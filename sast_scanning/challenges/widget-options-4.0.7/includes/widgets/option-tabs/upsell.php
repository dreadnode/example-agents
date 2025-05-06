<?php
/**
 * Upsell Extended Version Widget Options
 *
 * @copyright   Copyright (c) 2015, Jeffrey Carandang
 * @since       1.0
*/

// Exit if accessed directly
if ( ! defined( 'ABSPATH' ) ) exit;

/**
 * Add Upgrade to Pro Widget Options Tab
 *
 * @since 1.0
 * @return void
 */

 /**
 * Called on 'extended_widget_opts_tabs'
 * create new tab navigation for alignment options
 */
function widgetopts_tab_gopro( $args ){ ?>
    <li class="extended-widget-gopro-tab-alignment">
        <a href="#extended-widget-opts-tab-<?php echo $args['id'];?>-gopro">+</a>
    </li>
<?php
}
add_action( 'extended_widget_opts_tabs', 'widgetopts_tab_gopro', 100 );

/**
 * Called on 'extended_widget_opts_tabcontent'
 * create new tab content options for alignment options
 */
function widgetopts_tabcontent_gopro( $args ){ ?>
    <div id="extended-widget-opts-tab-<?php echo $args['id'];?>-gopro" class="extended-widget-opts-tabcontent extended-widget-opts-tabcontent-gopro">
        <p class="widgetopts-unlock-features">
            <span class="dashicons dashicons-lock"></span><?php _e( 'Unlock all Options', 'widget-options' );?>
        </p>
        <p>
            <?php _e( 'Get the world\'s most complete widget management and get the best out of your widgets! Upgrade to extended version to get:', 'widget-options' );?>
        </p>
        <ul>
            <li>
                <span class="dashicons dashicons-lock"></span> <?php _e( 'Animation Options', 'widget-options' );?>
            </li>
            <li>
                <span class="dashicons dashicons-lock"></span> <?php _e( 'Custom Styling Options', 'widget-options' );?>
            </li>
            <li>
                <span class="dashicons dashicons-lock"></span> <?php _e( 'Column Display', 'widget-options' );?>
            </li>
            <li>
                <span class="dashicons dashicons-lock"></span> <?php _e( 'User Roles Visibility Restriction', 'widget-options' );?>
            </li>
            <li>
                <span class="dashicons dashicons-lock"></span> <?php _e( 'Fixed/Sticky Widget Options', 'widget-options' );?>
            </li>
            <li>
                <span class="dashicons dashicons-lock"></span> <?php _e( 'Days and Date Range Restriction', 'widget-options' );?>
            </li>
            <li>
                <span class="dashicons dashicons-lock"></span> <?php _e( 'Link Widget Options', 'widget-options' );?>
            </li>
            <li>
                <span class="dashicons dashicons-lock"></span> <?php _e( 'Clone Widget Options', 'widget-options' );?>
            </li>
            <li>
                <span class="dashicons dashicons-lock"></span> <?php _e( 'Widget Caching Options', 'widget-options' );?>
            </li>
            <li>
                <span class="dashicons dashicons-lock"></span> <?php _e( 'Shortcodes Options', 'widget-options' );?>
            </li>
            <li>
                <span class="dashicons dashicons-lock"></span> <?php _e( 'Extended Taxonomy and Post Types Support', 'widget-options' );?>
            </li>
            <li>
                <span class="dashicons dashicons-lock"></span> <?php _e( 'Disable Widgets and Permissions', 'widget-options' );?>
            </li>
            <li>
                <span class="dashicons dashicons-lock"></span> <?php _e( 'Target URLs and Wildcard Restrictions', 'widget-options' );?>
            </li>
            <li>
                <span class="dashicons dashicons-lock"></span> <?php _e( 'Pagebuilder by SiteOrigin Support', 'widget-options' );?>
            </li>
        </ul>
        <p><strong><a href="<?php echo apply_filters('widget_options_site_url', trailingslashit(WIDGETOPTS_PLUGIN_WEBSITE).'?utm_source=wordpressadmin&utm_medium=widgettabs&utm_campaign=widgetoptsprotab');?>" class="button-primary" target="_blank"><?php _e( 'Learn More', 'widget-options' );?></a></strong></p>
    </div>
<?php
}
add_action( 'extended_widget_opts_tabcontent', 'widgetopts_tabcontent_gopro'); ?>
