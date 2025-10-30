<?php

/**
 * Handles additional widget tab options
 * run on __construct function
 */

// Exit if accessed directly
if (!defined('ABSPATH')) exit;


/**
 * Admin Messages
 * @return void
 */
if (!function_exists('widgetopts_admin_notices')) :
    function widgetopts_admin_notices()
    {
        if (!current_user_can('update_plugins'))
            return;

        //show rating notice to page that matters most
        global $pagenow;
        if (!in_array($pagenow, array('widgets.php', 'options-general.php'))) {
            return;
        }

        if ($pagenow == 'options-general.php' && function_exists('get_current_screen')) {
            $screen = get_current_screen();
            if (isset($screen->base) && $screen->base != 'settings_page_widgetopts_plugin_settings') {
                return;
            }
        }

        $install_date   = get_option('widgetopts_installDate');
        $saved          = get_option('widgetopts_RatingDiv');
        $display_date   = date('Y-m-d h:i:s');
        $datetime1      = new DateTime($install_date);
        $datetime2      = new DateTime($display_date);
        $diff_intrval   = round(($datetime2->format('U') - $datetime1->format('U')) / (60 * 60 * 24));
        if ('yes' != $saved && $diff_intrval >= 7) {
            echo '<div class="widgetopts_notice updated" style="box-shadow: 0 1px 1px 0 rgba(0,0,0,.1);">
            <p>Well done! You have been enjoying <strong>Widget Options</strong> for more than a week.
            <br>
            Do you love it? Are you over the moon? Will you give us a <a href="https://wordpress.org/support/view/plugin-reviews/widget-options" class="thankyou" target="_blank" title="Ok, you deserved it" style="font-weight:bold;"><strong>5-star rating</strong></a> on WordPress?
            </br>
            Your review is essential to the Widget Options community and our ongoing succes.
            <br><br>
            Thank you so much! � Your Widget Options Team
            <ul>
                <li><a href="https://wordpress.org/support/view/plugin-reviews/widget-options" class="thankyou" target="_blank" title="Ok, you deserved it" style="font-weight:bold;">' . __('Definitely. Widget Options is the best!', 'widget-options') . '</a></li>
                <li><a href="javascript:void(0);" class="widgetopts_bHideRating" title="I already did" style="font-weight:bold;">' . __('Already done!', 'widget-options') . '</a></li>
                <li><a href="https://widget-options.com/contact/" class="thankyou" target="_blank" title="Ok, you deserved it" style="font-weight:bold;">' . __("Not convinced yet. Still think about it.", 'widget-options') . '</a></li>
                <li><a href="javascript:void(0);" class="widgetopts_bHideRating" title="No, not good enough" style="font-weight:bold;">' . __("Dismiss", 'widget-options') . '</a></li>
            </ul>
        </div>
        <script>
        jQuery( document ).ready(function( $ ) {

        jQuery(\'.widgetopts_bHideRating\').click(function(){
            var data={\'action\':\'widgetopts_hideRating\'}
                 jQuery.ajax({

            url: "' . admin_url('admin-ajax.php') . '",
            type: "post",
            data: data,
            dataType: "json",
            async: !0,
            success: function(e) {
                if (e=="success") {
                   jQuery(\'.widgetopts_notice\').slideUp(\'slow\');

                }
            }
             });
            })

        });
        </script>
        ';
        }
    }
    add_action('admin_notices', 'widgetopts_admin_notices');
endif;

if (!function_exists('widgetopts_display_free_liecnse_admin_notice')) {
    /**
     * Show a notice to subscribe to newsletter
     */
    function widgetopts_display_free_liecnse_admin_notice()
    {
        $license_key = get_option('widgetopts_free_license');
        if (!current_user_can('update_plugins') || !empty($license_key))
            return;

        //show rating notice to page that matters most
        global $pagenow;
        if (!in_array($pagenow, array('options-general.php'))) {
            return;
        }

        if ($pagenow == 'options-general.php' && function_exists('get_current_screen')) {
            $screen = get_current_screen();
            if (isset($screen->base) && $screen->base != 'settings_page_widgetopts_plugin_settings') {
                return;
            }
        }

        $htmlNotice = '
            <div class="notice widgetopts-notice" style="border-left-color: #064466">
                <form method="post">
                    <h3>' . __( 'Free License', 'widget-options' ) .'</h3>
                    <p>' . __( "You're currently using the free version of Widget Options. To register a free license for the plugin, please fill in your email below. This is not required but helps us support you better.", 'widget-options' ) . '</p>
                    <input type="text" name="email" placeholder="' . __( 'Email Address', 'widget-options' ) . '" />
                    ' . wp_nonce_field( 'wo_free_license_action', 'wo_free_license_field' ) . '
                    <input type="submit" name="wo_free_license_activator" value="Register Free License" class="button button-primary" />
                    <input type="button" name="wo_free_license_dismiss" value="Dismiss" class="button button-secondary" /><br><br>
                    <input type="checkbox" name="wo_free_license_subscribe" value="1" checked /> Add me to your newsletter and keep me updated whenever you release news, updates and promos.
                    <p><small>* ' . __( 'Your email is secure with us! We will keep you updated on new feature releases and major announcements about Widget Options.', 'widget-options' ) . '</small></p>
                </form>
                <form method="post" id="wo_free_license_dismiss_form">
                    ' . wp_nonce_field( 'wo_free_license_dismiss', 'wo_free_license_dismiss_field' ) . '
                    <input type="hidden" name="wo_free_dismiss" value="1" />
                </form>
            </div>
            <script>
            jQuery( document ).ready(function( $ ) {

                jQuery(\'input[name="wo_free_license_dismiss"]\').on("click", function(e){
                    e.preventDefault();
                    jQuery("#wo_free_license_dismiss_form").submit();
                });

            });
            </script>
        ';

        echo $htmlNotice;
    }
    add_action('admin_notices', 'widgetopts_display_free_liecnse_admin_notice');

    function widgetopts_activate_free_liecnse () {
        if ( ! empty( $_POST['wo_free_license_activator'] ) ) {
            if ( ! wp_verify_nonce( sanitize_text_field( wp_unslash( $_POST[ 'wo_free_license_field' ] ) ), 'wo_free_license_action' ) ) {
                return;
            }

            $email = sanitize_email( wp_unslash( $_POST['email'] ) );

            if ( is_email( $email ) ) {
                $user       = get_user_by( 'email', $email );
                $first_name = '';
                $last_name  = '';
                $url        = rawurlencode( home_url() );

                if ( is_a( $user, 'WP_User' ) ) {
                    $first_name = $user->first_name;
                    $last_name  = $user->last_name;
                }

                if ( ! empty( $_POST['wo_free_license_subscribe'] ) ) {
                    // Make request, save key.
                    $request = wp_remote_post(
                        'https://widget-options.com/wp-admin/admin-ajax.php',
                        array(
                            'body'    =>
                                array(
                                    'action' => 'wo_free_license',
                                    'email_address' => $email,
                                    'fname'    => $first_name,
                                    'lname'     => $last_name,
                                    'url'           => $url,
                                )
                        )
                    );

                    if ( ! is_wp_error( $request ) ) {
                        $license = $email;

                        if ( ! empty( $license ) ) {
                            update_option( 'widgetopts_free_license', sanitize_text_field( $license ) );

                            add_action(
                                'admin_notices',
                                function() {
                                    ?>
                                    <div class="notice notice-success">
                                        <p><?php esc_html_e( 'Free license activated!', 'widget-options' ); ?></p>
                                    </div>
                                    <?php
                                }
                            );
                        }
                    } else {
                        add_action(
                            'admin_notices',
                            function() {
                                ?>
                                <div class="notice notice-error">
                                    <p><?php esc_html_e( 'Something went wrong! Try again later.', 'widget-options' ); ?></p>
                                </div>
                                <?php
                            }
                        );
                    }
                } else {
                    $license = $email;

                    if ( ! empty( $license ) ) {
                        update_option( 'widgetopts_free_license', sanitize_text_field( $license ) );

                        add_action(
                            'admin_notices',
                            function() {
                                ?>
                                <div class="notice notice-success">
                                    <p><?php esc_html_e( 'Free license activated!', 'widget-options' ); ?></p>
                                </div>
                                <?php
                            }
                        );
                    }
                }
            } else {
                add_action(
                    'admin_notices',
                    function() {
                        ?>
                        <div class="notice notice-error">
                            <p><?php esc_html_e( 'Invalid email address!', 'widget-options' ); ?></p>
                        </div>
                        <?php
                    }
                );
            }
        }
        if ( ! empty( $_POST['wo_free_dismiss'] ) ) {
            if ( ! wp_verify_nonce( sanitize_text_field( wp_unslash( $_POST[ 'wo_free_license_dismiss_field' ] ) ), 'wo_free_license_dismiss' ) ) {
                return;
            }

            update_option( 'widgetopts_free_license', sanitize_text_field( 'NA' ) );
        }
    }
    add_action( 'admin_init', 'widgetopts_activate_free_liecnse' );
}
