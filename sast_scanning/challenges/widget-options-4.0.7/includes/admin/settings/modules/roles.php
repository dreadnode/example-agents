<?php
/**
 * Roles Settings Module
 * Settings > Widget Options :: User Roles Restriction
 *
 * @copyright   Copyright (c) 2016, Jeffrey Carandang
 * @since       3.0
*/

// Exit if accessed directly
if ( ! defined( 'ABSPATH' ) ) exit;

/**
 * Create Card Module for User Roles Restriction
 *
 * @since 3.0
 * @global $widget_options
 * @return void
 */
if( !function_exists( 'widgetopts_settings_roles' ) ):
	function widgetopts_settings_roles(){ ?>
		<li class="widgetopts-module-card widgetopts-module-type-pro" data-module-id="roles">
			<div class="widgetopts-module-card-content">
				<a href="<?php echo apply_filters('widget_options_site_url', trailingslashit(WIDGETOPTS_PLUGIN_WEBSITE).'features/restrict-wordpress-widgets-per-user-roles/');?>" target="_blank" class="widgetopts-pro-upsell"></a>
				<h2><?php _e( 'User Roles Restriction', 'widget-options' );?></h2>
				<div class="widgetopts-pro-label"><span class="dashicons dashicons-lock"></span></div>
				<p class="widgetopts-module-desc">
					<?php _e( 'Restrict each widgets visibility for each user roles at ease via checkboxes.', 'widget-options' );?>
				</p>
			</div>
		</li>
	    <?php
	}
	add_action( 'widgetopts_module_cards', 'widgetopts_settings_roles', 100 );
endif;
?>
